package fi.helsinki.cs.probic.crypto

import java.security.KeyStore
import javax.crypto.Cipher
import org.apache.commons.codec.binary.Base64
import javax.security.cert.X509Certificate

/**
 * Mandatory options:
 * --intype hdfs or --intype socket
 * --input hdfs://path/to/input/folder
 * --output hdfs://path/to/output/foldersrc/main/scala/fi/helsinki/cs/probic/
 */
class PkCrypto(cert: String) {

  val keystorePath = "keystore.jks"

  lazy val password = scala.io.Source.fromFile("secret.txt", "UTF-8").getLines().toSeq.head.toCharArray

  lazy val decrypter = getDecrypt()

  lazy val key = getKey()

  def getKey() = {
    val ks = KeyStore.getInstance(KeyStore.getDefaultType())

    var fis: java.io.FileInputStream = null
    try {
      fis = new java.io.FileInputStream(keystorePath)
      ks.load(fis, password)
    } finally {
      if (fis != null) {
        fis.close();
      }
    }

    val k = ks.getKey(cert, password)
    k
  }

  /**
   * Method for testing only
   */
  def getCert(id: String) = {
    val ks = KeyStore.getInstance(KeyStore.getDefaultType())

    var fis: java.io.FileInputStream = null
    try {
      fis = new java.io.FileInputStream(keystorePath)
      ks.load(fis, password)
    } finally {
      if (fis != null) {
        fis.close();
      }
    }

    val k = ks.getCertificate(id)
    k
  }

  def getDecrypt() = {
    val rsa = Cipher.getInstance("RSA")
    rsa.init(Cipher.DECRYPT_MODE, key)
    rsa
  }

  def decrypt(cryptoText: Array[Byte]) = synchronized {
    new String(decrypter.doFinal(cryptoText))
  }

  def getRsa(pk: String) = {
    val publicBytes = Base64.decodeBase64(pk)
    val keySpec = X509Certificate.getInstance(publicBytes)
    val pubKey = keySpec.getPublicKey
    val rsa = Cipher.getInstance("RSA")
    rsa.init(Cipher.ENCRYPT_MODE, pubKey)
    rsa
  }

  def pkEncrypt(rsa: Cipher)(clearText: String) = {
    rsa.doFinal(clearText.getBytes)
  }

  /**
   * Prepare public key encryption engine for faster use. Used by test data generation and the clients sending the data.
   */
  def getEncrypter(id: String) = {
    val k = getCert(id).getEncoded
    val b64 = Base64.encodeBase64(k)
    val rsa = getRsa(new String(b64))
    pkEncrypt(rsa) _
  }
}
